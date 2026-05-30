package com.pq.agent_service.config;

import com.pq.agent_service.service.impl.IUserServiceImpl;
import com.pq.agent_service.utils.UserContext;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

@Slf4j
@Component
public class JwtInterceptor implements HandlerInterceptor {

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response,
                             Object handler) {
        String path = request.getRequestURI();
        log.info("[JwtInterceptor] path={}", path);
        if (path.contains("/users/") || path.contains("/error")) {
            log.info("[JwtInterceptor] skip auth for path={}", path);
            return true;
        }

        String authHeader = request.getHeader("Authorization");
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            log.warn("[JwtInterceptor] No valid Authorization header, return 401, path={}", path);
            response.setStatus(401);
            return false;
        }

        try {
            String token = authHeader.substring(7);
            Long userId = IUserServiceImpl.parseUserIdFromToken(token);
            String username = IUserServiceImpl.parseUsernameFromToken(token);
            UserContext.setUser(userId, username);
            return true;
        } catch (Exception e) {
            log.warn("JWT 解析失败: {}", e.getMessage());
            response.setStatus(401);
            return false;
        }
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response,
                                Object handler, Exception ex) {
        UserContext.removeUser();
    }
}
