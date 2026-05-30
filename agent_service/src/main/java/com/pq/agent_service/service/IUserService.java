package com.pq.agent_service.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pq.agent_service.domain.dto.UserDTO;
import com.pq.agent_service.domain.po.User;
import com.pq.agent_service.domain.vo.LoginVO;

public interface IUserService extends IService<User> {

    LoginVO register(UserDTO dto);

    LoginVO login(UserDTO dto);
}
